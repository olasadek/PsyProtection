import { useState } from 'react';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import PatientList from './components/PatientList';
import './styles/App.css';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedPatient, setSelectedPatient] = useState(null);

  return (
    <div className="app">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main>
        {activeTab === 'dashboard' ? (
          <Dashboard />
        ) : (
          <PatientList onSelectPatient={setSelectedPatient} />
        )}
      </main>
    </div>
  );
}

export default App;