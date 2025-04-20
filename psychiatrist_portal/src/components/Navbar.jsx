const Navbar = ({ activeTab, setActiveTab }) => {
    return (
      <nav className="navbar">
        <div className="logo">Drug Abuse Detection</div>
        <ul className="nav-links">
          <li 
            className={activeTab === 'dashboard' ? 'active' : ''}
            onClick={() => setActiveTab('dashboard')}
          >
            Dashboard
          </li>
          {/* <li 
            className={activeTab === 'patients' ? 'active' : ''}
            onClick={() => setActiveTab('patients')}
          >
            Patient History
          </li> */}
        </ul>
      </nav>
    );
  };
  
  export default Navbar;